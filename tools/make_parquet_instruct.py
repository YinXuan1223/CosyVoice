#!/usr/bin/env python3
"""Convert an existing CosyVoice `parquet/` shard dir into a `parquet_instruct/` dir
by adding an `instruct` column to every row of every shard.

parquet 和 parquet_instruct 的差別只有一個欄位: 
parquet_instruct 每一列多了一個'instruct' 欄位 (dtype=string)。
其餘欄位 (utt, audio_data, wav, text, spk, utt_embedding, spk_embedding, speech_token) 完全相同。

用法:
  python make_parquet_instruct.py --src_dir /home/ubuntu/TTS/Dataset/cosyvoice_lists_shuf/parquet --des_dir /home/ubuntu/TTS/Dataset/cosyvoice_lists_shuf/parquet_instruct

預設會替每一列填上和現有 parquet_instruct 資料一致的固定值:
    "You are a helpful assistant.<|endofprompt|>"

若你想針對每個 utt 給不同的 instruct 文字, 用 --instruct_file 指到一個
kaldi 風格的檔案 (每行 "utt instruct文字..."), 沒被列到的 utt 會 fallback 用
--instruct 的預設值。
"""
import argparse
import glob
import logging
import os

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DEFAULT_INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'


def load_instruct_file(path):
    utt2instruct = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt2instruct[parts[0]] = ' '.join(parts[1:])
    return utt2instruct


def convert_one(src_path, des_path, default_instruct, utt2instruct):
    df = pd.read_parquet(src_path)
    if utt2instruct:
        df['instruct'] = df['utt'].map(utt2instruct).fillna(default_instruct)
    else:
        df['instruct'] = default_instruct
    df.to_parquet(des_path)
    return des_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src_dir', type=str, default='/home/ubuntu/TTS/Dataset/cosyvoice_lists_shuf/parquet', help='existing parquet shard dir')
    parser.add_argument('--des_dir', type=str, default='/home/ubuntu/TTS/Dataset/cosyvoice_lists_shuf/parquet_instruct', help='output dir')
    parser.add_argument('--instruct', type=str, default=DEFAULT_INSTRUCT,
                         help='fixed instruct text applied to every row (default: matches existing data)')
    parser.add_argument('--instruct_file', type=str, default=None,
                         help='optional "utt instruct..." file for per-utt instruct text')
    parser.add_argument('--cv_shards', type=int, default=2,
                         help='number of trailing shards written into cv.data.list (0 to skip)')
    args = parser.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.src_dir, 'parquet_*.tar')))
    if not src_files:
        raise FileNotFoundError(f'no parquet_*.tar found under {args.src_dir}')

    os.makedirs(args.des_dir, exist_ok=True)
    utt2instruct = load_instruct_file(args.instruct_file) if args.instruct_file else None

    des_files = []
    for src_path in tqdm(src_files, desc='converting'):
        des_path = os.path.join(args.des_dir, os.path.basename(src_path))
        convert_one(src_path, des_path, args.instruct, utt2instruct)
        des_files.append(os.path.abspath(des_path))

    with open(os.path.join(args.des_dir, 'data.list'), 'w', encoding='utf8') as f:
        f.write('\n'.join(des_files) + '\n')

    if args.cv_shards > 0:
        with open(os.path.join(args.des_dir, 'cv.data.list'), 'w', encoding='utf8') as f:
            f.write('\n'.join(des_files[-args.cv_shards:]) + '\n')

    logging.info('done, wrote %d shards to %s', len(des_files), args.des_dir)


if __name__ == '__main__':
    main()
