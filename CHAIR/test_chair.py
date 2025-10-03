#!/usr/bin/env python3
import requests
import argparse
import json

def call_compute_caption(server_url, caption, image_id, timeout=30):
    payload = {"caption": caption, "image_id": image_id}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(server_url.rstrip('/') + '/computeCaption', json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def call_compute_caption_batch(server_url, items, timeout=30):
    headers = {"Content-Type": "application/json"}
    resp = requests.post(server_url.rstrip('/') + '/computeCaption', json=items, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='http://127.0.0.1:5000')
    parser.add_argument('--caption', type=str, default='A person sitting on a chair.')
    parser.add_argument('--image_id', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--batch_file', type=str, default='', help='optional json file containing {"captions": [...] } or a list')
    args = parser.parse_args()

    try:
        if args.batch_file:
            items = json.load(open(args.batch_file))
        else:
            items = [{"caption": args.caption, "image_id": int(args.image_id)}]

        res = call_compute_caption_batch(args.server, items, timeout=args.timeout)
        chair_s = res.get('CHAIRs') if isinstance(res, dict) else None
        chair_i = res.get('CHAIRi') if isinstance(res, dict) else None
        # DEBUG
        #print(json.dumps(res, indent=2, ensure_ascii=False))
        print("CHAIRi:", float(chair_i))
    except Exception as e:
        print('Request failed:', e)
        try:
            import sys
            sys.exit(1)
        except Exception:
            pass
