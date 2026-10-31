#!/usr/bin/env python3
"""Freeze or verify a reviewed Stage B interaction map without touching legacy records.

Source review is supplied explicitly; this CLI validates structure, not audiovisual truth.
Editing the timeline or map invalidates the bundle and dependent annotations.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.stage_b.structured import TimelineReview,InteractionMap,freeze_bundle,assert_frozen

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--verify',metavar='BUNDLE')
 p.add_argument('--timeline');p.add_argument('--map');p.add_argument('--roster')
 p.add_argument('--source-sha256');p.add_argument('--start-s',type=float);p.add_argument('--cutoff-s',type=float)
 p.add_argument('--reviewer');p.add_argument('--output')
 a=p.parse_args()
 if a.verify:
  assert_frozen(read(a.verify),TimelineReview.model_validate(read(a.timeline)) if a.timeline else None,InteractionMap.model_validate(read(a.map)) if a.map else None)
  print('Bundle structure and version match; human/source accuracy is not certified.');return
 if any(getattr(a,x) is None for x in ['timeline','map','roster','source_sha256','start_s','cutoff_s','reviewer','output']):p.error('freeze requires timeline, map, roster, source-sha256, start-s, cutoff-s, reviewer, output')
 b=freeze_bundle(InteractionMap.model_validate(read(a.map)),TimelineReview.model_validate(read(a.timeline)),read(a.roster),a.source_sha256,a.start_s,a.cutoff_s,a.reviewer)
 Path(a.output).write_text(json.dumps(b,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
