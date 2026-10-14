# -*- coding: utf-8 -*-
"""RA-10 confirmatory test on Section 2 (pre-registered): RUNNER. Computes the rows; the registered analysis
(protocolP.py, D30) is a separate step and is not run here.

Frozen specification, read before this file was written (nothing in it is changed here):
  00_PROJECT_CONTROL/RA10_SELECTION_RULE_v2.md (SHA-256 17311f07...edc8, D26); 00_PROJECT_CONTROL/DECISION_LOG.md
  D17, D18, D23-D30 (RA-00 freeze of 12 Oct 2026); 03_REANALYSIS/configurations/RA00_LEARNED_ARM_CANDIDATE.md Sec. 2-3.

One job = one S2 fold model (0-6) on one 20 m window of a Section 2 test piece (48 rows):
  window frame (D27, D29)  traces of the piece (LabelMe lines, 2 cm per pixel; x along the piece to the right, y down
                           the face) translated by (-x0, -y_offset) and clipped to [0, 20] x [0, H_window] (p2lib.clip).
                           S2B00P1: H_window 20 m centred on the 21.18 m image (y_offset 0.59 m), x0 = 0, 20, ..., 160 m.
                           S2B01P1: full image height (imageHeight x 0.02 = 17.94 m), x0 = 0, 20, 60, 80 m.
  realisations (D28)       run seeds (1337, 20260903, 7) x layouts; r = 3 x fold + seed index (21 per window and layout)
  layouts                  '5m' = p2v2.lines_layout('v', 5.0, 0.10, H) (primary); 'centre' = lines_layout('v', None, 0.10, H)
  observation (D29)        p2v2.observe(gt, lines, H, default_rng(run_v2.sd('EAST' + tag, int(x0), r, cname, 'obs')))
  LEARNED_ETP (D28, D29)   learned_arm.reconstruct(C, obs, lines, H, rng, load_models(fold, seed, 'rebuilt'),
                           evae_prior_nets(fold, H, x0, tag), 8 x seed index, dmc) with C = gast.prep_ctx('S2', fold)
                           and rng = default_rng(sd('EAST' + tag, int(x0), r, cname, 'LEARNED_ETP')); scored with
                           learned_arm.score_row (conventions of learned_arm.run_unseen_job)
  STAT_F, KDE_F, ADFNE_F, EVAE_F, RAW_EVAE, RAW_KDE, RAW_ADFNE (D28)
                           the arm code of the study's east_test.run_job, repeated line by line below, with fresh
                           fold-model networks east_test.nets_for(fold, H_window, x0) (piece tag through EPIECE, which
                           east_test reads at import) and arm key sd('EAST' + tag, int(x0), r, cname, arm)
  scores (D29)             Paper 1 statistical_tests.score with the S2 fold-0 cluster model, plus final_eval.p21map_err;
                           row keys scheme 'EAST' + tag, fold, box 'x' + int(x0) (at least two digits), cfg, arm, r

east_test.run_job and learned_arm.run_unseen_job are not called because their window is the full image height without
a vertical offset (the D27 window of S2B00P1 is a centred 20 m band). Every job runs in a fresh process (pool with
maxtasksperchild=1, or the 'single' mode), so east_test reads EPIECE, ELABELS and EX0S of that job at import.

Modes:
  python -B ra10s2_run.py verify                  recompute every frozen digest and stop on any difference
  python -B ra10s2_run.py single TAG X0 FOLD DIR  one job in this process; DIR/rows.csv, obs.csv, meta.json
  python -B ra10s2_run.py check                   determinism check: job (S2B01P1, x0 0, fold 0) twice in separate
                                                  processes; rows identical, LEARNED_ETP and STAT_F see the same hits
  python -B ra10s2_run.py full                    all 13 windows x 7 folds (16 workers, one fresh process per job)

Writes only into 03_REANALYSIS/results/RA10_S2 (ADFNE scratch files in its subfolder adfne_work, deleted by
study.adfne_batch after reading). The study, Paper 1 release and label folders are read only."""
import os, sys, json, time, math, hashlib, platform, subprocess, traceback
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

HERE = Path(__file__).resolve().parent                                      # 03_REANALYSIS/scripts
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'work' / 'RA10_S2'
CONTROL = PROJECT / 'work'
CONFIG = PROJECT / 'work'
RA32_MODELS = PROJECT / 'models' / 'termination'
STUDY = PROJECT / 'study'
REL = PROJECT
LABEL_DIR = PROJECT / 'work' / 'not_released'
ADFNE_ROOT = Path(os.environ.get('ADFNE_ROOT', PROJECT / 'work' / 'adfne'))
OCTAVE = Path(os.environ.get('OCTAVE', 'octave-cli'))
os.environ['ADFNE_ROOT'] = str(ADFNE_ROOT)
os.environ['ADFNE_WORK'] = str(OUT / 'adfne_work')

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

VERSION = 'ra10s2_run 1.0 (2026-10-12)'
SEEDS = (1337, 20260903, 7)
NFOLD = 7
ARMS = ('LEARNED_ETP', 'STAT_F', 'KDE_F', 'ADFNE_F', 'EVAE_F', 'RAW_EVAE', 'RAW_KDE', 'RAW_ADFNE')
COMP_ARMS = ARMS[1:]
LAYOUTS = ('5m', 'centre')
KEY = ['scheme', 'fold', 'box', 'cfg', 'arm', 'r']
DET_JOB = ('S2B01P1', 0.0, 0)
NWORKERS = 16

# windows (D27; piece tags D29). H None = full image height (imageHeight x 0.02, as east_test.east_traces)
PIECES = {
    'S2B00P1': dict(name='S2_B00_RL0849-0876_P1', labels=LABEL_DIR / 'S2_B00_RL0849-0876_P1.json',
                    x0s=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0), H=20.0, y_offset=0.59,
                    image_height_m=21.18, n_traces=80, d27_pieces=(14, 10, 8, 8, 13, 8, 16, 12, 10),
                    labels_sha256='0aabcb036c792ed224deada5fab9e00d6e7396af7b9b361cdc331711c83de1bb'),
    'S2B01P1': dict(name='S2_B01_RL0876-0903_P1', labels=LABEL_DIR / 'S2_B01_RL0876-0903_P1.json',
                    x0s=(0.0, 20.0, 60.0, 80.0), H=None, y_offset=0.0,
                    image_height_m=17.94, n_traces=65, d27_pieces=(15, 13, 11, 25),
                    labels_sha256='74023fdb89438b95949f296830ec74a5002b6bc350b9b5bc5f2c1e860055e2a2'),
}

# frozen digests: RA00_LEARNED_ARM_CANDIDATE.md Sec. 4.1-4.3, DECISION_LOG D26-D28, RA00_FREEZE_LIST_DRAFT.md Sec. 1-2
EXPECTED_CODE = {
    HERE / 'learned_arm.py': '1bfe37f9c7d498c487ec10048c030e8472e5fe0c217d178ee5e179e9ae3e8269',
    STUDY / 'east_test.py': 'a37cb3035c946329fb30116c4d484a2cec817b35d56020128fb4c57de9fa6976',
    STUDY / 'eval_learned.py': '6943b17ec2fa811cfcdc67f512b23a1ec5f7d3e21411cde199e10bfff1f15158',
    STUDY / 'final_eval.py': '1377ab3ab642c4e088f45c32dbbf6f1b93427476f5a9fffbcc97aa423ef080e2',
    STUDY / 'gast.py': '9965d03b1881706e6033041f5ecfc3d42153229713ac6ec21d94f7eddb8cd0d2',
    STUDY / 'learned.py': '6dd04a72cad158b6c164014bb60bd135b7aa967eea59f80b0b6b83dad5ee1f6f',
    STUDY / 'p2lib.py': 'd193c31563b39322093a85262ce374d81fb3c36b5d2084f4e9183273f5c8b135',
    STUDY / 'p2v2.py': 'c22a1e241dd0824520767237a5cf1440bee283da0e8f850e535f08ab721657fb',
    STUDY / 'rules2.py': '0a2341beb53a6e8ec0aba67aec9de1ce533a68757e8866533345156ee5772a66',
    STUDY / 'run_ab.py': '07f55d4a02a2940496b4f7523ba647f4a22fbfa5e8c6aa0c6e501112dc859cba',
    STUDY / 'run_dd.py': '1f4f1e0ecfec4a9f5d5390f5ca00d5353b615f0f11f78b3fa81e3c87d5aa7fc3',
    STUDY / 'run_extra.py': '85d7c2b33bd75ed2c2349d182aad56e5bf4b4930985b439d8add3f5d821f9677',
    STUDY / 'run_tests.py': '961ae2e2e8df7b12ff4ac311aaf496484a6a4e3d1126a8d610341ef9187cac5b',
    STUDY / 'run_v2.py': 'ccf54042c72f9670a7f600c4c5fc6abff36a8e2ac040deabe1bb9f79391e7e66',
    STUDY / 'run_v3_term.py': '4cb7e559dd4dd7fde7acd7b4e23edc6f464f6f5aa1f2249580dc83c8c2f21826',
    STUDY / 'tr_common.py': '9f036b1282d38c17f6181def6080269e07fc66e16bc057d33b28afae887f210c',
    STUDY / 'tr_gen.py': '206d692ce519cc9b027ddc575acb4b4d67a3ec6f7b30c6d11d300244f3aa1712',
    STUDY / 'tr_tip.py': 'df4b13d183780bed504dd76247266a458f11e7e98dd739a0de64a834f9bb6b50',
    REL / 'scripts' / 'statistical_tests.py': '93fdb07b1896c9b0fced9fe3aff4c314b287703c960054ae7bc477e98e98682f',
    REL / 'src' / 'evaluation' / 'common.py': 'caa8bd0aa300b32329d29ec966b091d23eabf8e7ac93d12c4a10661248e3b004',
    REL / 'src' / 'evaluation' / 'generators.py': 'd64a6728f63d191ddb1f90d19df2ff1192b40bfe530145c14b8158784298f15e',
    REL / 'src' / 'model' / 'training.py': 'b4ff7b54d38f840f97b6f8c0a98993f1d732d9dd9d88b9386a323d8aadd3f816',
    REL / 'src' / 'model' / 'utils.py': '018c7434aa3cb0deae763fe93c206ff0cb25523129a15b402b02eed9a88c3d9e',
    REL / 'src' / 'model' / 'utils_additions.py': 'ca55e58f6e2166d416b56178c24377ae38911246b8038b2ff5482cd397f34248',
    REL / 'src' / 'repo_paths.py': '3c05b1df2d746d4a5e80ea6694cdd7f625fb6b98dcf4feebc3215f3b8c4d6324',
    REL / 'src' / 'study' / 'setbg.py': '36bfc8b80b6bc573b65f813d323bf96f28c9dfccfda5b3a9d3dc1095fd4c8ef2',
    REL / 'src' / 'study' / 'study.py': 'b27d8c573c1cf6e26577ebbcb502ba9e2920a44ead4d41327c455a23e4233cf8',
    REL / 'src' / 'study' / 'topology.py': '1b0c8aaaefd35a7969e54e02e7475038a125c0530a8e69474ccd5d1c6e785a01',
    ADFNE_ROOT / 'run_adfne.bat': '2fe81c3d00f6308db617186db1d4478d31b6be5f1d59b35a4f5113a8b990ab01',
    ADFNE_ROOT / 'adfne_init.m': 'dcab034b42581a61f34ee06ab744aa10c667bc7d1fc453fef8e3851f283ebff9',
    ADFNE_ROOT / 'octave_compat' / 'setstructfields.m': 'e394415da05b2fffc3dea8f6c548a8b84f49040067bd39f50b9f446ef8c98b19',
    OCTAVE: '387ac823fb5b01b3407fb34edda6ab1cdba0c9619841a0f6dee323d05a850f2d',
}
EXPECTED_ABBREV = {HERE / 'protocolP.py': ('51c04bb1', 'c919')}          # D28 and D30 quote the abbreviated digest
EXPECTED_DATA = {
    CONTROL / 'RA10_SELECTION_RULE_v2.md': '17311f07abf1ca99031c85b9102d12e6927b0c3d00ac4f868300273d4146edc8',
    REL / 'data' / 'orientation_clusters' / 'S2_f0.json': 'f8713563edf2563c24751ca6c2c7676bc96d110c66775b8a38616283fb1070ec',
    PIECES['S2B00P1']['labels']: PIECES['S2B00P1']['labels_sha256'],
    PIECES['S2B01P1']['labels']: PIECES['S2B01P1']['labels_sha256'],
}
TERM2 = {
    (0, 1337): '0816c4872d76e4d97526007e194d3bc5aa1470abd1f34e4f7dee50d5ab4503f4',
    (0, 20260903): 'd36a417a3474670674b01602611918a80716fe7d5bbb2ff79f2e2d14992b4b64',
    (0, 7): '9f696b2341ed79419e1d4c9519027b76eaf2da4f8b0551748939b3eb45ed600e',
    (1, 1337): '2f53e2f5c3dacc6bbb363fed9c5257f4b3d9a913f0458add7d84e5ad7739d939',
    (1, 20260903): '639e6bb9393496dd6e362a13f4cfb0d61ac1d05be28a8703d1515cf3eba533bf',
    (1, 7): 'a04bb4b20744f92d0fcb3e1e201ff1d679439aeef59dc7fadea485c0dd9d015b',
    (2, 1337): '810b7a96a14b1f4029ff8c66fe910bcddc707a5d3d098ccbac0107268e21639c',
    (2, 20260903): 'e709754270773e1be9ce1a59c1b0495c7c4db1e2bb9d088ab6bd2c492b383944',
    (2, 7): '58aef0b2ffe8983e7876de15e7b191967f1b272bea484d9065bc9ffd6a18f194',
    (3, 1337): 'ae8ce51358ff888b9c13faba8b59076b45061eda2899d0c6a91cf9f01994e022',
    (3, 20260903): '6e9531cd99d7631db402fe343ae55a1675959ac4d84921177e2277be9d3e7641',
    (3, 7): 'af0cfaa9c84ddcf3d36e0ca6274430e5755fc1b396c005ecfd5123203674779a',
    (4, 1337): 'dfb50e7ba9b772e79519995603f19984d1f2dba4d4334f252db57bda6bc66035',
    (4, 20260903): 'c275be58979b5574067695586d9f7f8660a5ebfe73f2b60afed45bd649fadf97',
    (4, 7): '856006c2be6b85cf989424a62ea40497a8e4bf03a017df407bb922cb7f6a23c1',
    (5, 1337): 'ed5470feac087cc46d36076805f2d8cf091d7b73cd7969de18aef3a67b23035b',
    (5, 20260903): '963189aae066fc46957f07827802d48842513f4e0df206577ec13ee6ad43dc95',
    (5, 7): 'a2b50576b17d07e6d6f9cf8cecd455fbda79dbe7a4a3423cbda6a8c16134a0da',
    (6, 1337): 'bc101dacaeaefb8a1dfcd969646ae126541a7eb6350a144734ff126debd6027d',
    (6, 20260903): '4ca98b44f701729d3081f9d703accc201b3e9866d8cdf998b3997b52f1cf9940',
    (6, 7): '2606572c992e827e7d65c8ddfb7c3c40e4ab0a98fba0eb9109f14e4a2c2159b2',
}


def families():
    """file families and their frozen combined digests (ANALYSIS_MANIFEST.md Sec. 1 rule)"""
    fs = [(f, s) for f in range(NFOLD) for s in SEEDS]
    return {
        'rebuilt termination networks term2_S2_f*_s*.pt (RA-32)': ([RA32_MODELS / f'term2_S2_f{f}_s{s}.pt' for f, s in fs],
                                                                  '73a9bcedd5623dec19d740cf1f6f11b1ade4030532c14adee27ae291c4c4e35f'),
        'sequential transformer variant 2, study tr_models/gast2_S2_f*_s*.pt': ([STUDY / 'tr_models' / f'gast2_S2_f{f}_s{s}.pt' for f, s in fs],
                                                                                '1e8bc469eb309855e6e2d1cd75ba96032ab954aa6fbf98f18843365137a5a79c'),
        'evidence networks, study tr_models/evn_S2_f*_s*.pt': ([STUDY / 'tr_models' / f'evn_S2_f{f}_s{s}.pt' for f, s in fs],
                                                               '2124fdcdeceed53ea9a7bd31d41be1e2a5f03397d545db629ecbb4c164c282d2'),
        'Paper 1 EVAE checkpoints, release models/evae/S2_f*_s*/checkpoint.pt': ([REL / 'models' / 'evae' / f'S2_f{f}_s{s}' / 'checkpoint.pt' for f, s in fs],
                                                                                  '026af582582785a80c5100149eacba354176963dd7052ca790259b38ae25fd3e'),
        'release box library (manifest + 50 windows)': ([REL / 'data' / 'box_library' / 'box_manifest.csv'] + sorted((REL / 'data' / 'box_library' / 'boxes').glob('*.json')),
                                                        '38c20caf22a47ae973f8687691540156ff831228613f14c3a32ba62ca2d6d377'),
        'release splits': (sorted((REL / 'data' / 'splits').glob('*.json')), '2a91795ef04a6174351dd3c550b2d0a620f58f04c88b425102ac130137d5a727'),
        'release orientation clusters': (sorted((REL / 'data' / 'orientation_clusters').glob('*.json')),
                                         '4557691d68f75fec9dd33a6379e6c045a25a12e2b5048b7f82ec0b92c159d1ed'),
        'ADFNE 1.5 clone (adfne_benchmark_2026-08-21/ADFNE1.5)': (local_walk(ADFNE_ROOT / 'ADFNE1.5'),
                                                                  'b90ee9e2dbcbecb051983dc6d583234ea162f24b7756c3fdf15b90763898116e'),
    }


FAMILY_COUNTS = {'release box library (manifest + 50 windows)': 51, 'release splits': 15, 'release orientation clusters': 15,
                 'ADFNE 1.5 clone (adfne_benchmark_2026-08-21/ADFNE1.5)': 87}


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def combined(files):
    """ANALYSIS_MANIFEST.md Sec. 1 rule: sorted by full path; update with the file name, then its hex digest"""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: str(x)):
        h.update(Path(f).name.encode()); h.update(sha256(f).encode())
    return h.hexdigest()


def local_walk(root):
    """all files below a LOCAL folder (version-control folders skipped); never used on local storage"""
    if 'onedrive' in str(root).lower():
        raise RuntimeError('local_walk must not be used on local storage')
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ('.git', '.svn', '.hg')]
        out += [Path(dp) / f for f in fn]
    return sorted(out, key=str)


def imported_files():
    """files of every imported module that lives in the study folder, the Paper 1 release or the project scripts"""
    roots = [str(STUDY).lower(), str(REL).lower(), str(HERE).lower()]
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if not f or not os.path.isabs(f) or not os.path.exists(f):
            continue
        fl = str(Path(f).resolve()).lower()
        if any(fl.startswith(r) for r in roots) and fl.endswith('.py'):
            p = str(Path(f).resolve())
            out[p] = dict(module=name, bytes=os.path.getsize(p), sha256=sha256(p))
    return out


def environment():
    import numpy, scipy, pandas
    d = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__,
             platform=platform.platform(), executable=sys.executable, cpu_count=os.cpu_count(), pid=os.getpid(),
             dont_write_bytecode=sys.dont_write_bytecode, flag_B=bool(sys.flags.dont_write_bytecode),
             CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), PYTHONDONTWRITEBYTECODE=os.environ.get('PYTHONDONTWRITEBYTECODE'),
             OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'), MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
             ADFNE_ROOT=os.environ.get('ADFNE_ROOT'), ADFNE_WORK=os.environ.get('ADFNE_WORK'))
    if 'torch' in sys.modules:
        import torch
        d.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available())
    return d


def obs_digest(obs):
    """SHA-256 of the full hit list (aperture, orientation, point, trace direction, trace index, aperture limits; exact floats)"""
    h = hashlib.sha256()
    for o in obs:
        h.update((f"{o['ap']}|{o['o']}|{float(o['x']).hex()}|{float(o['y']).hex()}|{float(o['th']).hex()}|{int(o['tid'])}|"
                  f"{float(o['lo']).hex()}|{float(o['hi']).hex()};").encode())
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=str)


# ------------------------------------------------------------------ digest verification (freeze, D28 and candidate Sec. 6 item 6)
def verify(step):
    """recompute every frozen digest; returns the record (all_ok False = STOP)"""
    started = now(); checks = []

    def add(kind, name, got, expected, ok, note=''):
        checks.append(dict(kind=kind, file=str(name), sha256=got, expected=expected, ok=bool(ok), note=note))

    for f, e in EXPECTED_CODE.items():
        g = sha256(f) if f.exists() else 'MISSING'; add('code', f, g, e, g == e)
    for f, (a, b) in EXPECTED_ABBREV.items():
        g = sha256(f) if f.exists() else 'MISSING'; add('code', f, g, f'{a}...{b}', g.startswith(a) and g.endswith(b), 'frozen digest quoted abbreviated (D28, D30)')
    for f, e in EXPECTED_DATA.items():
        g = sha256(f) if f.exists() else 'MISSING'; add('data', f, g, e, g == e)
    for (fi, s), e in TERM2.items():
        f = RA32_MODELS / f'term2_S2_f{fi}_s{s}.pt'; g = sha256(f) if f.exists() else 'MISSING'; add('model', f, g, e, g == e)
    for name, (files, e) in families().items():
        missing = [str(f) for f in files if not Path(f).exists()]
        if missing:
            add('family', name, 'MISSING', e, False, f'missing files: {missing[:3]}'); continue
        g = combined(files); n_ok = len(files) == FAMILY_COUNTS.get(name, 21)
        add('family', name, g, e, g == e and n_ok, f'{len(files)} files')
    # label files: content as frozen in D27 (fracture lines only, two points each, no no_data polygons)
    for tag, sp in PIECES.items():
        d = json.load(open(sp['labels'], encoding='utf-8'))
        labs = [s['label'] for s in d['shapes']]
        ok = (labs.count('fracture') == sp['n_traces'] and len(labs) == sp['n_traces']
              and all(len(s['points']) == 2 for s in d['shapes'])
              and abs(d['imageHeight'] * 0.02 - sp['image_height_m']) < 1e-9)
        add('labels', sp['labels'], f"{labs.count('fracture')} fracture lines, {len(labs) - labs.count('fracture')} other shapes, "
            f"image {d['imageWidth']} x {d['imageHeight']} px", f"{sp['n_traces']} fracture lines, image height {sp['image_height_m']} m", ok)
        if sp['H'] is not None:     # centred window band
            off = (d['imageHeight'] * 0.02 - sp['H']) / 2
            add('window', tag, f'centred offset {off!r}', f"y_offset {sp['y_offset']}", abs(off - sp['y_offset']) < 1e-9)
    rec = dict(step=f'verify ({step})', version=VERSION, started=started, finished=now(), runner_sha256=sha256(Path(__file__).resolve()),
               all_ok=all(c['ok'] for c in checks), n_checks=len(checks), n_failed=sum(not c['ok'] for c in checks), checks=checks,
               environment=environment())
    write_json(OUT / f'run_record_verify_{step}.json', rec)
    print(f"verify ({step}): {rec['n_checks']} checks, {rec['n_failed']} failed", flush=True)
    for c in checks:
        if not c['ok']:
            print('  FAILED', c['kind'], c['file'], c['sha256'], 'expected', c['expected'], c['note'], flush=True)
    return rec


# ------------------------------------------------------------------ one job
def run_job(job):
    """one S2 fold on one window: 8 arms x 3 run seeds x 2 layouts = 48 rows. Returns dict(rows, obs, meta)."""
    tag, x0, fi = job
    x0 = float(x0); fi = int(fi)
    sp = PIECES[tag]
    started = now(); t_job = time.time()
    # read by east_test at import; this process runs only this job (pool maxtasksperchild=1, or the 'single' mode)
    env_set = dict(EPIECE=tag, ELABELS=str(sp['labels']), EX0S=','.join(str(v) for v in sp['x0s']), ETAG='ra10s2_' + tag,
                   EARMS=','.join(COMP_ARMS))
    os.environ.update(env_set)
    import warnings, collections
    wcount = collections.Counter()

    def _show(message, category, filename, lineno, file=None, line=None):
        first = str(message).splitlines()[0] if str(message) else ''
        wcount[f'{category.__name__}: {first[:180]} ({Path(str(filename)).name}:{lineno})'] += 1
    out = dict(rows=[], obs=[], meta=dict(tag=tag, x0=x0, fold=fi, started=started, env_set=env_set))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = _show
            _job_body(tag, x0, fi, sp, out)
    except Exception:
        out['meta']['error'] = traceback.format_exc()
    out['meta'].update(finished=now(), job_sec=time.time() - t_job, warnings=dict(wcount), environment=environment(),
                       code=imported_files())
    return out


def _job_body(tag, x0, fi, sp, out):
    import torch
    torch.set_num_threads(1)                                                 # as east_test.run_job and run_unseen_job
    import learned_arm as LA                                                 # frozen learned arm (puts study + release on sys.path)
    import east_test as ET                                                   # comparator arms (reads EPIECE etc. now)
    if ET.PIECE != tag or Path(ET.LABELS) != Path(sp['labels']):
        raise RuntimeError(f'east_test read EPIECE={ET.PIECE!r}, ELABELS={ET.LABELS!s}; expected {tag!r}, {sp["labels"]}')
    import p2lib as P
    import p2v2 as V
    import final_eval as FE
    import gast as G
    from run_v2 import sd
    import statistical_tests as ST
    assert tuple(P.SEEDS) == SEEDS
    meta = out['meta']

    L, Himg = LA.piece_traces(sp['labels'])                                  # metres, 2 cm per pixel (east_test.east_traces)
    H = float(sp['H']) if sp['H'] is not None else Himg                      # window height (D27)
    yoff = float(sp['y_offset'])
    gt = P.clip(L - np.array([x0, yoff, x0, yoff]), H)                       # window frame (D29)
    ym = H / 20
    scheme = 'EAST' + tag; box = f'x{int(x0):02d}'
    meta.update(scheme=scheme, box=box, H=H, y_offset=yoff, image_height_m=Himg, n_traces_piece=int(len(L)), gt_n=int(len(gt)),
                gt_len_m=float(P.seglen(gt).sum()), n_pieces_ge_0_5m=int((P.seglen(gt) >= 0.5).sum()),
                d27_pieces=int(sp['d27_pieces'][list(sp['x0s']).index(x0)]), adfne_tag=f'east{tag}_f{fi}_x{int(x0)}')
    smodel = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / 'S2_f0.json')       # one scoring model for all arms

    # comparator arms: context and fresh networks exactly as east_test.run_job
    C = G.prep_ctx('S2', fi)
    t0 = time.time(); nets = ET.nets_for(fi, H, x0); meta['nets_for_sec'] = time.time() - t0
    # learned arm: context and EVAE prior realisations exactly as learned_arm.run_unseen_job
    C_l = G.prep_ctx('S2', fi)
    ev_l = LA.evae_prior_nets(fi, H, x0, tag)
    meta['evae_prior_equals_nets_for'] = bool(len(ev_l) == len(nets['EVAE']) == 24 and
                                              all(np.array_equal(a, b) for a, b in zip(ev_l, nets['EVAE'])))
    meta['n_networks'] = {k: len(v) for k, v in nets.items()}
    meta['models'] = []
    dmc, dmc_l = {}, {}
    rows, obsrows = out['rows'], out['obs']
    for si, seed in enumerate(P.SEEDS):
        models = LA.load_models(fi, seed, 'rebuilt')                         # gast2 + evidence network + rebuilt TermNet2
        meta['models'].append(dict(seed=seed, mode=models['mode'], term=str(LA.term_path(fi, seed, 'rebuilt')),
                                   gnet=str(G.MODELS / f'gast2_S2_f{fi}_s{seed}.pt'), evn=str(G.MODELS / f'evn_S2_f{fi}_s{seed}.pt')))
        r = fi * 3 + si                                                      # 21 realisations
        for spacing in (5.0, None):
            cname = '5m' if spacing else 'centre'
            lines = V.lines_layout('v', spacing, FE.BASE['APERTURE'], H)
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, 'obs')))
            dg = obs_digest(obs); dg_l = None
            for arm in ARMS:
                if arm == 'LEARNED_ETP':                                     # learned_arm.run_unseen_job, lines 344-348
                    obs_l = V.observe(gt, lines, H, np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, 'obs')))
                    dg_l = obs_digest(obs_l)
                    rng = np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, arm)); t0 = time.time()
                    rec = LA.reconstruct(C_l, obs_l, lines, H, rng, models, ev_l, si * 8, dmc_l)
                    rows.append(LA.score_row(rec, gt, H, smodel, scheme, fi, box, cname, arm, r, time.time() - t0))
                    continue
                # east_test.run_job, lines 86-115 (arms STAT_F, KDE_F, ADFNE_F, EVAE_F, RAW_*)
                rng = np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, arm)); t0 = time.time()
                if arm.startswith('RAW_'):
                    rec = nets[arm[4:]][si * 8]
                else:
                    if arm == 'STAT_F':
                        pf = lambda k: P.unif_proposals(60, C['tr_th'], C['tr_len3d'], H, rng)      # noqa: E731
                    else:
                        src = nets[arm[:-2]]; pf = lambda k, src=src: src[(si * 8 + k) % 24]       # noqa: E731
                    rec = FE.frozen(C, obs, lines, H, rng, pf, FE.BASE, dmc, True)
                s = ST.score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, ym, smodel)
                s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                                 map_rmse=FE.p21map_err(rec, gt, H), sec=time.time() - t0, **s))
            obsrows.append(dict(scheme=scheme, box=box, fold=fi, r=r, cfg=cname, n_hits_comparators=len(obs), n_hits_learned=len(obs_l),
                                hits_sha256_comparators=dg, hits_sha256_learned=dg_l, identical=bool(dg == dg_l),
                                hits=' '.join(f"{o['ap']}:{o['tid']}" for o in obs)))
    meta['dmin_cache'] = dict(comparators={str(k): v for k, v in dmc.items()}, learned={str(k): v for k, v in dmc_l.items()})


# ------------------------------------------------------------------ modes
def single(tag, x0, fi, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    res = run_job((tag, float(x0), int(fi)))
    res['meta']['command'] = ' '.join([sys.executable] + sys.argv); res['meta']['runner_sha256'] = sha256(Path(__file__).resolve())
    write_json(outdir / 'meta.json', res['meta'])
    if 'error' in res['meta']:
        print(res['meta']['error'], flush=True); raise SystemExit(2)
    pd.DataFrame(res['rows']).to_csv(outdir / 'rows.csv', index=False, float_format='%.17g')
    pd.DataFrame(res['obs']).to_csv(outdir / 'obs.csv', index=False)
    print(f"single {tag} x0 {x0} fold {fi}: {len(res['rows'])} rows, {res['meta']['job_sec']:.0f} s", flush=True)


def compare_rows(a, b):
    """bitwise comparison of two row tables matched on KEY (sec excluded); NaN equals NaN"""
    a = a.sort_values(KEY, kind='mergesort').reset_index(drop=True); b = b.sort_values(KEY, kind='mergesort').reset_index(drop=True)
    same_keys = len(a) == len(b) and bool((a[KEY].astype(str).values == b[KEY].astype(str).values).all())
    cols = [c for c in a.columns if c not in KEY + ['sec']]
    same_cols = list(a.columns) == list(b.columns)
    if not (same_keys and same_cols):
        return dict(rows_a=len(a), rows_b=len(b), same_keys=same_keys, same_columns=same_cols, all_equal=False)
    A = a[cols].to_numpy(dtype=float); B = b[cols].to_numpy(dtype=float)
    eq = (A == B) | (np.isnan(A) & np.isnan(B))
    fin = np.isfinite(A) & np.isfinite(B)
    md = float(np.max(np.abs(A - B)[fin])) if fin.any() else 0.0
    diff_cols = [c for j, c in enumerate(cols) if not eq[:, j].all()]
    return dict(rows_a=len(a), rows_b=len(b), same_keys=same_keys, same_columns=same_cols, n_values=int(eq.size),
                n_equal=int(eq.sum()), max_abs_diff=md, differing_columns=diff_cols, all_equal=bool(eq.all()))


def check():
    v = verify('check')
    if not v['all_ok']:
        raise SystemExit('STOP: a frozen digest differs (see run_record_verify_check.json); nothing was run')
    started = now(); t0 = time.time(); det = OUT / 'determinism'; runs = []
    for i in (1, 2):                                                         # sequential: the two runs share ADFNE file names
        d = det / f'run{i}'
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'single', DET_JOB[0], repr(DET_JOB[1]), str(DET_JOB[2]), str(d)]
        t = time.time(); p = subprocess.run(cmd, env=dict(os.environ)); runs.append(dict(command=' '.join(cmd), returncode=p.returncode, sec=time.time() - t))
        if p.returncode != 0:
            raise SystemExit(f'STOP: determinism run {i} failed (return code {p.returncode})')
    r1 = pd.read_csv(det / 'run1' / 'rows.csv', float_precision='round_trip'); r2 = pd.read_csv(det / 'run2' / 'rows.csv', float_precision='round_trip')
    o1 = pd.read_csv(det / 'run1' / 'obs.csv'); o2 = pd.read_csv(det / 'run2' / 'obs.csv')
    m1 = json.load(open(det / 'run1' / 'meta.json')); m2 = json.load(open(det / 'run2' / 'meta.json'))
    cmp_ = compare_rows(r1, r2)
    ok_cols = ['scheme', 'box', 'fold', 'r', 'cfg', 'n_hits_comparators', 'n_hits_learned', 'hits_sha256_comparators', 'hits_sha256_learned', 'hits']
    obs_between = bool(o1[ok_cols].astype(str).equals(o2[ok_cols].astype(str)))
    obs_within = bool(o1['identical'].all() and o2['identical'].all())
    evae_same = bool(m1['evae_prior_equals_nets_for'] and m2['evae_prior_equals_nets_for'])
    n_expected = len(ARMS) * len(SEEDS) * len(LAYOUTS)
    PASS = bool(cmp_['all_equal'] and len(r1) == len(r2) == n_expected and obs_between and obs_within and evae_same
                and len(o1) == len(SEEDS) * len(LAYOUTS))
    lines_ = [f'RA-10 Section 2 determinism check ({started}); runner {VERSION}',
              f'job: piece {DET_JOB[0]}, x0 {DET_JOB[1]:g} m, S2 fold {DET_JOB[2]}; run twice in separate processes (`single` mode), one after the other',
              f'rows per run: {len(r1)} and {len(r2)} (expected {n_expected} = {len(ARMS)} arms x {len(SEEDS)} run seeds x {len(LAYOUTS)} layouts)',
              f"rows matched by key: {cmp_.get('same_keys')}; same columns: {cmp_.get('same_columns')}",
              f"values compared (all columns except the keys and sec): {cmp_.get('n_values')}; bitwise equal: {cmp_.get('n_equal')}; "
              f"largest absolute difference: {cmp_.get('max_abs_diff')}",
              f'hit lists of LEARNED_ETP equal to those of STAT_F and the other comparator arms (same window, r, layout): '
              f'run 1 {int(o1.identical.sum())} of {len(o1)}, run 2 {int(o2.identical.sum())} of {len(o2)}',
              f'hit lists equal between the two runs: {obs_between}',
              f'EVAE prior realisations of the learned arm (learned_arm.evae_prior_nets) equal to east_test.nets_for EVAE: run 1 '
              f"{m1['evae_prior_equals_nets_for']}, run 2 {m2['evae_prior_equals_nets_for']}",
              f"process ids: run 1 {m1['environment']['pid']}, run 2 {m2['environment']['pid']}; torch threads in the job: "
              f"{m1['environment'].get('torch_threads')}, {m2['environment'].get('torch_threads')}",
              f"job seconds: {m1['job_sec']:.0f}, {m2['job_sec']:.0f}",
              'RESULT: PASS (identical rows; identical observations)' if PASS else 'RESULT: FAIL', f'runtime {time.time() - t0:.0f} s']
    txt = '\n'.join(lines_); print(txt, flush=True)
    (det / 'determinism_check_log.txt').write_text(txt + '\n', encoding='utf-8')
    write_json(OUT / 'run_record_check.json', dict(step='check', version=VERSION, command=' '.join([sys.executable] + sys.argv),
                                                    started=started, finished=now(), runner_sha256=sha256(Path(__file__).resolve()),
                                                    runs=runs, comparison=cmp_, obs_within=obs_within, obs_between=obs_between,
                                                    evae_same=evae_same, PASS=PASS, environment=environment()))
    if not PASS:
        raise SystemExit('STOP: determinism check failed')


def full():
    v = verify('full')
    if not v['all_ok']:
        raise SystemExit('STOP: a frozen digest differs (see run_record_verify_full.json); nothing was run')
    ck = OUT / 'run_record_check.json'
    if not ck.exists() or not json.load(open(ck))['PASS']:
        raise SystemExit('STOP: the determinism check has not passed')
    from multiprocessing import Pool
    jobs = [(tag, x0, fi) for tag, sp in PIECES.items() for x0 in sp['x0s'] for fi in range(NFOLD)]
    started = now(); t0 = time.time(); R, O, M, errors = [], [], [], []
    print(f'full run: {len(jobs)} jobs, {NWORKERS} workers, one process per job', flush=True)
    with Pool(NWORKERS, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            m = res['meta']; M.append(m)
            if 'error' in m:
                errors.append(m); print(f"ERROR {m['tag']} x0 {m['x0']:g} fold {m['fold']}:\n{m['error']}", flush=True)
            else:
                R += res['rows']; O += res['obs']
            print(f"{i + 1}/{len(jobs)} {m['tag']} x0 {m['x0']:g} fold {m['fold']} {m.get('job_sec', 0):.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)
            if (i + 1) % 10 == 0:
                pd.DataFrame(R).to_csv(OUT / 'rows_RA10_S2.partial.csv', index=False, float_format='%.17g')
    finished = now(); sec = time.time() - t0
    df = pd.DataFrame(R)
    if len(df):
        order_tag = {t: i for i, t in enumerate(PIECES)}; order_arm = {a: i for i, a in enumerate(ARMS)}; order_cfg = {c: i for i, c in enumerate(LAYOUTS)}
        sk = pd.DataFrame(dict(a=df.scheme.str[4:].map(order_tag), b=df.box.str[1:].astype(int), c=df.fold, d=df.r, e=df.cfg.map(order_cfg), f=df.arm.map(order_arm)))
        df = df.loc[sk.sort_values(list('abcdef'), kind='mergesort').index].reset_index(drop=True)
    name = 'rows_RA10_S2.csv' if not errors else 'rows_RA10_S2.INCOMPLETE.csv'
    df.to_csv(OUT / name, index=False, float_format='%.17g')
    ob = pd.DataFrame(O)
    if len(ob):
        ob = ob.sort_values(['scheme', 'box', 'fold', 'r', 'cfg'], kind='mergesort').reset_index(drop=True)
    ob.to_csv(OUT / 'obs_check_RA10_S2.csv', index=False)
    part = OUT / 'rows_RA10_S2.partial.csv'
    if part.exists():
        part.unlink()
    # cross-check: the determinism job inside the pool run equals the two separate-process runs
    cross = None
    det1 = OUT / 'determinism' / 'run1' / 'rows.csv'
    if det1.exists() and len(df):
        back = pd.read_csv(OUT / name, float_precision='round_trip')
        sel = back[(back.scheme == 'EAST' + DET_JOB[0]) & (back.box == f'x{int(DET_JOB[1]):02d}') & (back.fold == DET_JOB[2])]
        cross = compare_rows(sel, pd.read_csv(det1, float_precision='round_trip'))
    rec = dict(step='full', version=VERSION, command=' '.join([sys.executable] + sys.argv), started=started, finished=finished, seconds=sec,
               runner_sha256=sha256(Path(__file__).resolve()), workers=NWORKERS, jobs=len(jobs), errors=len(errors), rows=len(df),
               rows_file=name, cross_check_with_determinism_run=cross, environment=environment(), job_meta=M)
    write_json(OUT / 'run_record_full.json', rec)
    print(f'full run finished: {len(df)} rows, {len(errors)} failed jobs, {sec:.0f} s; written {name}', flush=True)
    if cross is not None:
        print('cross-check with the determinism run:', 'identical' if cross['all_equal'] else cross, flush=True)
    if errors:
        raise SystemExit('STOP: some jobs failed (see run_record_full.json)')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'verify':
        rv = verify('verify')
        raise SystemExit(0 if rv['all_ok'] else 'STOP: a frozen digest differs')
    elif mode == 'single':
        single(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif mode == 'check':
        check()
    elif mode == 'full':
        full()
    else:
        print(__doc__)
