#!/usr/bin/env python

#################
# ETL2
# http://etlcdb.db.aist.go.jp/etlcdb/etln/form_k.htm
# http://etlcdb.db.aist.go.jp/etlcdb/etln/etl2/e2code.jpg
#################

import numpy as np
import os
import CO59_kanadata

files_dir = os.path.dirname(__file__)
etl_dir = os.path.join(files_dir, "./ETL2")
files = ["ETL2_1","ETL2_2","ETL2_3","ETL2_4","ETL2_5",]
out_dir = os.path.join(files_dir, "./ETL2_npy")


BYTES_IN_RECORD = 2745

width, height = 60, 60
out_size = 30
bpp = 6 # bits per pixel

def is_hiragana(bits):
    if bits[5] == 0:
        return False 
    if bits[0] == 1:
        return False

def is_katakana(bits):
    if bits[5] == 0:
        return False
    if bits[0] == 0:
        return False

def bits_to_image(bits):
    # add (8-bpp) bits for padding
    bits = bits.reshape(-1, bpp)
    pad = np.array([0]*bits.shape[0]*(8-bpp), dtype=np.uint8).reshape(-1, 8-bpp)
    bits = np.hstack([pad, bits])
    
    # float image
    data = np.packbits(bits).astype(np.float32).reshape([width, height])
    max = np.power(2, bpp)-1
    data = (data / max *2)-1
    # resize to half x half
    data = (data[::2, ::2] + data[1::2, ::2] + data[::2, 1::2] + data[1::2, 1::2])/4

    return data

def read_logical_record(record):
    bits = np.unpackbits(record).reshape([-1, 6])
    code = (int(np.packbits(bits[28]))/4, int(np.packbits(bits[29]))/4)

    if code in CO59_kanadata.hiragana:
        code = "hira_"+CO59_kanadata.hiragana[code]
        # image starts on 61th character(=45th byte)
        image_bits = np.unpackbits(record[45:])
        return code, bits_to_image(image_bits)
    elif code in CO59_kanadata.katakana:
        code = "kata_"+CO59_kanadata.katakana[code]
        # image starts on 61th character(=45th byte)
        image_bits = np.unpackbits(record[45:])
        return code, bits_to_image(image_bits)
    else:
         return code, None

def read_files(files):
    dic = {}
    for f in files:
        path = os.path.join(etl_dir, f)
        if not os.path.exists(path):
            print("Not found:", path)
            continue
        print("Open:", path)
        rows = np.fromfile(path, dtype=np.uint8).reshape([-1, BYTES_IN_RECORD])
        for record in rows:
            code, image = read_logical_record(record)
            if image == None:
                continue
            if not code in dic:
                dic[code] = np.array([], dtype=np.float32)
            dic[code] = np.append(dic[code], image)
    return dic


dic = read_files(files)
zure = [
    "kata_A", "kata_I", "kata_U", "kata_E", "kata_O",
    "kata_KA", "kata_KI", "kata_KU", "kata_KE", "kata_KO",
    "kata_SA", "kata_SI"
]
for k,v in dic.items():
    images = v.reshape([-1, out_size, out_size])
    if k in zure:
        images = images[1:]
        idx = zure.index(k)
        if idx < len(zure)-1:
            zurekomi = dic[zure[idx+1]].reshape([-1, out_size, out_size])[0:1]
            images = np.append(images, zurekomi, axis=0)
    
    print(k, images.shape[0])
    path = os.path.join(out_dir, k)
    np.save(path, images)